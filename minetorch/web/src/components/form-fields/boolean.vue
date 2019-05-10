<template>
  <el-form-item :label="label">
    <el-switch v-model="formData[name]" />
  </el-form-item>
</template>
<script>
import formFieldMixin from 'components/form-fields/form-field'

export default {
  mixins: [formFieldMixin],
}
</script>
<style lang="scss">
</style>
